/**
 * Global Store
 *
 * Central state management for the workspace.
 * Combines entity stores for different domain objects.
 *
 * @module @neurectomy/state
 * @author @APEX @PRISM
 */

import type { Subscription, Disposable, DockPosition } from "../interfaces";
import {
  Entity,
  EntityStore,
  MutableObservable,
  generateEntityId,
  type EntityDefinition,
} from "./Entity";

// ============================================================================
// Workspace State Types
// ============================================================================

/**
 * Panel entity definition
 */
export interface PanelEntity extends EntityDefinition {
  id: string;
  name: string;
  position: DockPosition;
  isVisible: boolean;
  size: number;
  isCollapsed: boolean;
  badge?: number | string;
  order: number;
}

/**
 * Pane entity definition
 */
export interface PaneEntity extends EntityDefinition {
  id: string;
  itemIds: string[];
  activeItemId: string | null;
  isZoomed: boolean;
  activationHistory: string[];
}

/**
 * Item entity definition
 */
export interface ItemEntity extends EntityDefinition {
  id: string;
  type: string;
  title: string;
  path?: string;
  isDirty: boolean;
  isPreview: boolean;
  isPinned: boolean;
  paneId: string;
}

/**
 * Project entity definition
 */
export interface ProjectEntity extends EntityDefinition {
  id: string;
  rootPath: string;
  name: string;
  folders: string[];
}

/**
 * Layout entity definition
 */
export interface LayoutEntity extends EntityDefinition {
  id: string;
  type: "leaf" | "split";
  direction?: "horizontal" | "vertical";
  paneId?: string;
  children?: string[];
  sizes?: number[];
  parentId?: string;
}

/**
 * Global workspace state
 */
export interface WorkspaceGlobalState {
  activePaneId: string | null;
  focusedElement: "pane" | "dock" | "panel" | null;
  focusedDockPosition: DockPosition | null;
  sidebarVisible: boolean;
  sidebarPosition: "left" | "right";
  panelVisible: boolean;
  statusBarVisible: boolean;
  zenMode: boolean;
  theme: "light" | "dark" | "system";
}

// ============================================================================
// Global Store
// ============================================================================

/**
 * Central store managing all workspace state
 */
export class GlobalStore implements Disposable {
  // Entity stores
  readonly panels = new EntityStore<PanelEntity>();
  readonly panes = new EntityStore<PaneEntity>();
  readonly items = new EntityStore<ItemEntity>();
  readonly projects = new EntityStore<ProjectEntity>();
  readonly layout = new EntityStore<LayoutEntity>();

  // Global state
  readonly globalState = new MutableObservable<WorkspaceGlobalState>({
    activePaneId: null,
    focusedElement: null,
    focusedDockPosition: null,
    sidebarVisible: true,
    sidebarPosition: "left",
    panelVisible: true,
    statusBarVisible: true,
    zenMode: false,
    theme: "dark",
  });

  // Dock visibility (quick access)
  readonly leftDockVisible = new MutableObservable<boolean>(true);
  readonly rightDockVisible = new MutableObservable<boolean>(false);
  readonly bottomDockVisible = new MutableObservable<boolean>(true);

  // Active panel per dock
  readonly leftDockActivePanel = new MutableObservable<string | null>(null);
  readonly rightDockActivePanel = new MutableObservable<string | null>(null);
  readonly bottomDockActivePanel = new MutableObservable<string | null>(null);

  private disposed = false;

  // ─────────────────────────────────────────────────────────────────────────
  // Panel Operations
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Register a panel
   */
  registerPanel(config: Omit<PanelEntity, "id">): Entity<PanelEntity> {
    const id = generateEntityId("panel");
    return this.panels.create({ ...config, id });
  }

  /**
   * Toggle panel visibility
   */
  togglePanel(panelId: string): void {
    const panel = this.panels.get(panelId);
    if (!panel) return;

    const state = panel.get();
    panel.update({ isVisible: !state.isVisible });

    // Update dock active panel
    if (!state.isVisible) {
      this.setActivePanelForDock(state.position, panelId);
    }
  }

  /**
   * Set active panel for a dock
   */
  setActivePanelForDock(position: DockPosition, panelId: string | null): void {
    switch (position) {
      case "left":
        this.leftDockActivePanel.set(panelId);
        break;
      case "right":
        this.rightDockActivePanel.set(panelId);
        break;
      case "bottom":
        this.bottomDockActivePanel.set(panelId);
        break;
    }
  }

  /**
   * Get panels by position
   */
  getPanelsByPosition(position: DockPosition): Entity<PanelEntity>[] {
    return this.panels.query((entity) => entity.get().position === position);
  }

  /**
   * Get visible panels by position
   */
  getVisiblePanelsByPosition(position: DockPosition): Entity<PanelEntity>[] {
    return this.panels.query((entity) => {
      const state = entity.get();
      return state.position === position && state.isVisible;
    });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Pane Operations
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Create a new pane
   */
  createPane(): Entity<PaneEntity> {
    const id = generateEntityId("pane");
    const pane = this.panes.create({
      id,
      itemIds: [],
      activeItemId: null,
      isZoomed: false,
      activationHistory: [],
    });

    // Set as active if first pane
    if (this.panes.size === 1) {
      this.setActivePane(id);
    }

    return pane;
  }

  /**
   * Set the active pane
   */
  setActivePane(paneId: string): void {
    this.globalState.update((state) => ({
      ...state,
      activePaneId: paneId,
      focusedElement: "pane",
    }));
  }

  /**
   * Get the active pane
   */
  getActivePane(): Entity<PaneEntity> | undefined {
    const { activePaneId } = this.globalState.get();
    return activePaneId ? this.panes.get(activePaneId) : undefined;
  }

  /**
   * Add item to pane
   */
  addItemToPane(paneId: string, itemId: string, activate = true): void {
    const pane = this.panes.get(paneId);
    if (!pane) return;

    const state = pane.get();
    const itemIds = [...state.itemIds, itemId];
    const updates: Partial<PaneEntity> = { itemIds };

    if (activate) {
      updates.activeItemId = itemId;
      updates.activationHistory = [
        itemId,
        ...state.activationHistory.filter((id) => id !== itemId),
      ];
    }

    pane.update(updates);
  }

  /**
   * Remove item from pane
   */
  removeItemFromPane(paneId: string, itemId: string): void {
    const pane = this.panes.get(paneId);
    if (!pane) return;

    const state = pane.get();
    const itemIds = state.itemIds.filter((id) => id !== itemId);
    const activationHistory = state.activationHistory.filter(
      (id) => id !== itemId
    );

    let activeItemId = state.activeItemId;
    if (activeItemId === itemId) {
      // Activate next item from history or first remaining
      activeItemId = activationHistory[0] ?? itemIds[0] ?? null;
    }

    pane.update({ itemIds, activeItemId, activationHistory });
  }

  /**
   * Set active item in pane
   */
  setActiveItem(paneId: string, itemId: string): void {
    const pane = this.panes.get(paneId);
    if (!pane) return;

    const state = pane.get();
    if (!state.itemIds.includes(itemId)) return;

    const activationHistory = [
      itemId,
      ...state.activationHistory.filter((id) => id !== itemId),
    ];

    pane.update({ activeItemId: itemId, activationHistory });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Item Operations
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Create a new item
   */
  createItem(config: Omit<ItemEntity, "id">): Entity<ItemEntity> {
    const id = generateEntityId("item");
    return this.items.create({ ...config, id });
  }

  /**
   * Find item by path (for deduplication)
   */
  findItemByPath(path: string): Entity<ItemEntity> | undefined {
    return this.items.find((entity) => entity.get().path === path);
  }

  /**
   * Get items for a pane
   */
  getItemsForPane(paneId: string): Entity<ItemEntity>[] {
    return this.items.query((entity) => entity.get().paneId === paneId);
  }

  /**
   * Update item dirty state
   */
  setItemDirty(itemId: string, isDirty: boolean): void {
    const item = this.items.get(itemId);
    item?.update({ isDirty });
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Layout Operations
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Initialize layout with single pane
   */
  initializeLayout(): Entity<LayoutEntity> {
    const pane = this.createPane();
    const id = generateEntityId("layout");
    return this.layout.create({
      id,
      type: "leaf",
      paneId: pane.id,
    });
  }

  /**
   * Split a pane
   */
  splitPane(
    paneId: string,
    direction: "horizontal" | "vertical"
  ): Entity<PaneEntity> | null {
    const paneLayout = this.layout.find(
      (entity) => entity.get().paneId === paneId
    );
    if (!paneLayout) return null;

    // Create new pane
    const newPane = this.createPane();

    // Create split layout
    const splitId = generateEntityId("layout");
    const originalLeafId = generateEntityId("layout");
    const newLeafId = generateEntityId("layout");

    // Create leaf for original pane
    this.layout.create({
      id: originalLeafId,
      type: "leaf",
      paneId,
      parentId: splitId,
    });

    // Create leaf for new pane
    this.layout.create({
      id: newLeafId,
      type: "leaf",
      paneId: newPane.id,
      parentId: splitId,
    });

    // Update original layout to be a split
    paneLayout.update({
      type: "split",
      direction,
      paneId: undefined,
      children: [originalLeafId, newLeafId],
      sizes: [50, 50],
    });

    return newPane;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Focus Operations
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Focus a pane
   */
  focusPane(paneId: string): void {
    this.globalState.update((state) => ({
      ...state,
      activePaneId: paneId,
      focusedElement: "pane",
      focusedDockPosition: null,
    }));
  }

  /**
   * Focus a dock
   */
  focusDock(position: DockPosition): void {
    this.globalState.update((state) => ({
      ...state,
      focusedElement: "dock",
      focusedDockPosition: position,
    }));
  }

  /**
   * Focus a panel
   */
  focusPanel(panelId: string): void {
    const panel = this.panels.get(panelId);
    if (!panel) return;

    const position = panel.get().position;
    this.globalState.update((state) => ({
      ...state,
      focusedElement: "panel",
      focusedDockPosition: position,
    }));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // UI State Operations
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Toggle sidebar visibility
   */
  toggleSidebar(): void {
    this.globalState.update((state) => ({
      ...state,
      sidebarVisible: !state.sidebarVisible,
    }));
  }

  /**
   * Toggle panel (bottom dock) visibility
   */
  toggleBottomPanel(): void {
    const current = this.bottomDockVisible.get();
    this.bottomDockVisible.set(!current);
  }

  /**
   * Toggle zen mode
   */
  toggleZenMode(): void {
    this.globalState.update((state) => {
      const zenMode = !state.zenMode;
      return {
        ...state,
        zenMode,
        sidebarVisible: !zenMode,
        panelVisible: !zenMode,
        statusBarVisible: !zenMode,
      };
    });
  }

  /**
   * Set theme
   */
  setTheme(theme: "light" | "dark" | "system"): void {
    this.globalState.update((state) => ({ ...state, theme }));
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Serialization
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Serialize the entire store state
   */
  serialize(): object {
    return {
      version: 1,
      globalState: this.globalState.get(),
      panels: this.panels.getAll().map((e) => e.get()),
      panes: this.panes.getAll().map((e) => e.get()),
      items: this.items.getAll().map((e) => e.get()),
      projects: this.projects.getAll().map((e) => e.get()),
      layout: this.layout.getAll().map((e) => e.get()),
      docks: {
        left: {
          visible: this.leftDockVisible.get(),
          activePanel: this.leftDockActivePanel.get(),
        },
        right: {
          visible: this.rightDockVisible.get(),
          activePanel: this.rightDockActivePanel.get(),
        },
        bottom: {
          visible: this.bottomDockVisible.get(),
          activePanel: this.bottomDockActivePanel.get(),
        },
      },
    };
  }

  /**
   * Restore state from serialized data
   */
  deserialize(data: ReturnType<typeof this.serialize>): void {
    // Implementation would restore all entity states
    // For now, just log that we would restore
    console.log("Would restore state:", data);
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Cleanup
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Dispose all stores and subscriptions
   */
  dispose(): void {
    if (!this.disposed) {
      this.disposed = true;
      this.panels.dispose();
      this.panes.dispose();
      this.items.dispose();
      this.projects.dispose();
      this.layout.dispose();
      this.globalState.dispose();
      this.leftDockVisible.dispose();
      this.rightDockVisible.dispose();
      this.bottomDockVisible.dispose();
      this.leftDockActivePanel.dispose();
      this.rightDockActivePanel.dispose();
      this.bottomDockActivePanel.dispose();
    }
  }
}

// ============================================================================
// Global Store Singleton
// ============================================================================

let globalStoreInstance: GlobalStore | null = null;

/**
 * Get or create the global store instance
 */
export function getGlobalStore(): GlobalStore {
  if (!globalStoreInstance) {
    globalStoreInstance = new GlobalStore();
  }
  return globalStoreInstance;
}

/**
 * Reset the global store (for testing)
 */
export function resetGlobalStore(): void {
  globalStoreInstance?.dispose();
  globalStoreInstance = null;
}
