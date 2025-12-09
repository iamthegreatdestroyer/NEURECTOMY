/**
 * Item Interface
 *
 * Defines the contract for items that can be displayed in panes (tabs).
 * Based on Zed's Item trait adapted for TypeScript/React.
 *
 * Items are the content shown in editor panes - files, terminals,
 * previews, settings, etc.
 *
 * @module @neurectomy/interfaces
 * @author @APEX @ARCHITECT
 */

import type { ReactNode } from "react";
import type { LucideIcon } from "lucide-react";
import type {
  EntityId,
  FocusHandle,
  Disposable,
  Subscription,
  CancellationToken,
  Result,
} from "./types";

// ============================================================================
// Item Identity
// ============================================================================

/**
 * Item ID type for type-safe item identification
 */
export type ItemId = EntityId<"item">;

/**
 * Item type identifiers
 */
export type ItemType =
  | "editor"
  | "terminal"
  | "preview"
  | "settings"
  | "welcome"
  | "diff"
  | "image"
  | "webview"
  | "output"
  | "search-results"
  | "custom";

// ============================================================================
// Tab Presentation
// ============================================================================

/**
 * Information for rendering an item's tab
 */
export interface TabContent {
  /** Tab title */
  title: string;

  /** Short title for narrow tabs */
  shortTitle?: string;

  /** Tab icon */
  icon?: LucideIcon;

  /** Tab icon color (CSS color string) */
  iconColor?: string;

  /** Whether item has unsaved changes */
  isDirty: boolean;

  /** Whether tab is in preview mode (italic, single click open) */
  isPreview: boolean;

  /** Whether tab is pinned */
  isPinned: boolean;

  /** Tooltip text */
  tooltip?: string;

  /** Badge text or number */
  badge?: string | number;

  /** Badge color variant */
  badgeVariant?: "default" | "info" | "warning" | "error";

  /** Description shown in tab detail */
  description?: string;
}

/**
 * Full tab state including presentation and behavior
 */
export interface TabState extends TabContent {
  /** Item ID this tab represents */
  itemId: ItemId;

  /** Item type */
  type: ItemType;

  /** Whether tab can be closed */
  canClose: boolean;

  /** Whether tab can be dragged */
  canDrag: boolean;

  /** Whether tab is currently active */
  isActive: boolean;
}

// ============================================================================
// Project Path
// ============================================================================

/**
 * Project path for workspace-relative items
 */
export interface ProjectPath {
  /** Workspace root (for multi-root workspaces) */
  workspaceId: string;

  /** Relative path from workspace root */
  relativePath: string;

  /** Full absolute path */
  absolutePath: string;
}

// ============================================================================
// Item Events
// ============================================================================

/**
 * Event fired when item state changes
 */
export interface ItemChangeEvent {
  /** Type of change */
  type: "content" | "dirty" | "title" | "icon" | "saved";

  /** The item that changed */
  itemId: ItemId;

  /** Additional change data */
  data?: unknown;
}

/**
 * Event fired when item requests navigation
 */
export interface ItemNavigationEvent {
  /** Target path or URI */
  target: string;

  /** Whether to open in new tab */
  newTab: boolean;

  /** Whether to preview */
  preview: boolean;

  /** Position to navigate to */
  position?: { line: number; character: number };
}

// ============================================================================
// Item Interface (Core)
// ============================================================================

/**
 * Core item interface - the primary contract for all tab items.
 *
 * This is analogous to Zed's Item trait:
 * ```rust
 * pub trait Item: FocusableView + EventEmitter<Self::Event> + Render {
 *     type Event;
 *     fn tab_content(&self, params: TabContentParams, cx: &WindowContext) -> AnyElement;
 *     fn tab_tooltip_text(&self, cx: &AppContext) -> Option<SharedString>;
 *     fn tab_description(&self, detail: usize, cx: &AppContext) -> Option<SharedString>;
 *     fn project_path(&self, cx: &AppContext) -> Option<ProjectPath>;
 *     fn is_dirty(&self, cx: &AppContext) -> bool;
 *     fn has_conflict(&self, cx: &AppContext) -> bool;
 *     fn can_save(&self, cx: &AppContext) -> bool;
 *     fn save(...) -> Task<Result<()>>;
 *     fn can_save_as(...) -> bool;
 *     fn save_as(...) -> Task<Result<()>>;
 *     fn reload(...) -> Task<Result<()>>;
 *     fn clone_on_split(...) -> Option<...>;
 *     fn is_singleton(...) -> bool;
 *     fn pixel_position_of_cursor(...) -> Option<Point<Pixels>>;
 * }
 * ```
 */
export interface Item extends Disposable {
  // ─────────────────────────────────────────────────────────────────────────
  // Identity
  // ─────────────────────────────────────────────────────────────────────────

  /** Unique item identifier */
  readonly id: ItemId;

  /** Item type for serialization and UI */
  readonly type: ItemType;

  // ─────────────────────────────────────────────────────────────────────────
  // Tab Content (Zed: tab_content)
  // ─────────────────────────────────────────────────────────────────────────

  /** Get current tab content for rendering */
  getTabContent(): TabContent;

  /** Subscribe to tab content changes */
  onTabContentChange(callback: (content: TabContent) => void): Subscription;

  // ─────────────────────────────────────────────────────────────────────────
  // Project Path (Zed: project_path)
  // ─────────────────────────────────────────────────────────────────────────

  /** Get project path if this item is workspace-relative */
  getProjectPath(): ProjectPath | null;

  /** Get the unique identifier for deduplication (e.g., file path) */
  getDeduplicationKey(): string | null;

  // ─────────────────────────────────────────────────────────────────────────
  // Dirty State (Zed: is_dirty, has_conflict)
  // ─────────────────────────────────────────────────────────────────────────

  /** Whether item has unsaved changes */
  isDirty(): boolean;

  /** Whether item has merge conflict or external changes */
  hasConflict(): boolean;

  /** Subscribe to dirty state changes */
  onDirtyChange(callback: (isDirty: boolean) => void): Subscription;

  // ─────────────────────────────────────────────────────────────────────────
  // Persistence (Zed: can_save, save, save_as, reload)
  // ─────────────────────────────────────────────────────────────────────────

  /** Whether item can be saved */
  canSave(): boolean;

  /** Save item to its current location */
  save(token?: CancellationToken): Promise<Result<void>>;

  /** Whether item can be saved to a new location */
  canSaveAs(): boolean;

  /** Save item to a new location */
  saveAs(path: string, token?: CancellationToken): Promise<Result<void>>;

  /** Reload item from disk */
  reload(token?: CancellationToken): Promise<Result<void>>;

  // ─────────────────────────────────────────────────────────────────────────
  // Splitting (Zed: clone_on_split, is_singleton)
  // ─────────────────────────────────────────────────────────────────────────

  /** Whether item can be split (opened in multiple panes) */
  canSplit(): boolean;

  /** Create a clone for displaying in another pane */
  cloneForSplit(): Item | null;

  /** Whether only one instance of this item should exist */
  isSingleton(): boolean;

  // ─────────────────────────────────────────────────────────────────────────
  // Focus
  // ─────────────────────────────────────────────────────────────────────────

  /** Focus handle for keyboard navigation */
  readonly focusHandle: FocusHandle;

  /** Request focus to this item */
  focus(): void;

  /** Check if item has focus */
  hasFocus(): boolean;

  // ─────────────────────────────────────────────────────────────────────────
  // Rendering
  // ─────────────────────────────────────────────────────────────────────────

  /** Render the item content */
  render(): ReactNode;

  /** Render item-specific toolbar actions */
  renderToolbar?(): ReactNode;

  /** Render breadcrumb navigation */
  renderBreadcrumb?(): ReactNode;

  // ─────────────────────────────────────────────────────────────────────────
  // Lifecycle
  // ─────────────────────────────────────────────────────────────────────────

  /** Called when item becomes active in a pane */
  onActivate?(): void;

  /** Called when item becomes inactive */
  onDeactivate?(): void;

  /** Called when item is about to be closed - return false to prevent */
  onWillClose?(): boolean | Promise<boolean>;

  /** Called when item is closed */
  onClose?(): void;

  // ─────────────────────────────────────────────────────────────────────────
  // Events
  // ─────────────────────────────────────────────────────────────────────────

  /** Subscribe to item change events */
  onChange(callback: (event: ItemChangeEvent) => void): Subscription;

  /** Subscribe to navigation requests */
  onNavigate(callback: (event: ItemNavigationEvent) => void): Subscription;

  // ─────────────────────────────────────────────────────────────────────────
  // Serialization
  // ─────────────────────────────────────────────────────────────────────────

  /** Serialize item state for persistence */
  serialize(): ItemSerializedState | null;
}

// ============================================================================
// Item Serialization
// ============================================================================

/**
 * Serialized item state for workspace persistence
 */
export interface ItemSerializedState {
  /** Item type for deserialization */
  type: ItemType;

  /** Type-specific state data */
  data: unknown;

  /** Version for migration */
  version: number;
}

/**
 * Deserializer function signature
 */
export type ItemDeserializer = (
  state: ItemSerializedState
) => Promise<Item | null>;

// ============================================================================
// Item Factory
// ============================================================================

/**
 * Item creation parameters
 */
export interface ItemCreateParams {
  /** Optional ID (generated if not provided) */
  id?: ItemId;

  /** Initial state data */
  data?: unknown;

  /** Whether to open in preview mode */
  preview?: boolean;
}

/**
 * Factory function for creating item instances
 */
export type ItemFactory<T extends Item = Item> = (
  params: ItemCreateParams
) => T;

/**
 * Item registration entry
 */
export interface ItemRegistration<T extends Item = Item> {
  /** Item type identifier */
  type: ItemType;

  /** Factory function */
  factory: ItemFactory<T>;

  /** Deserializer function */
  deserialize: ItemDeserializer;

  /** File extensions this item handles (for file items) */
  extensions?: string[];

  /** MIME types this item handles */
  mimeTypes?: string[];

  /** Priority for handling (higher = preferred) */
  priority?: number;
}

// ============================================================================
// Item Registry Interface
// ============================================================================

/**
 * Central registry for all item types in the workspace
 */
export interface ItemRegistry {
  /** Register a new item type */
  register<T extends Item>(registration: ItemRegistration<T>): Disposable;

  /** Unregister an item type */
  unregister(type: ItemType): void;

  /** Get item registration */
  get(type: ItemType): ItemRegistration | undefined;

  /** Get registration for file extension */
  getForExtension(extension: string): ItemRegistration | undefined;

  /** Get registration for MIME type */
  getForMimeType(mimeType: string): ItemRegistration | undefined;

  /** Get all registered item types */
  getAll(): readonly ItemRegistration[];

  /** Create an item instance */
  create(type: ItemType, params?: ItemCreateParams): Item | undefined;

  /** Deserialize an item */
  deserialize(state: ItemSerializedState): Promise<Item | null>;
}

// ============================================================================
// Item Handle (for Pane management)
// ============================================================================

/**
 * Handle for managing items within panes.
 * Similar to Zed's ItemHandle trait for type-erased item access.
 */
export interface ItemHandle {
  /** Get the item ID */
  readonly id: ItemId;

  /** Get the item type */
  readonly type: ItemType;

  /** Get tab state for rendering */
  getTabState(): TabState;

  /** Get the underlying item (downcast) */
  getItem(): Item;

  /** Activate this item in its pane */
  activate(): void;

  /** Close this item */
  close(): Promise<boolean>;

  /** Pin/unpin this item */
  setPinned(pinned: boolean): void;

  /** Set preview mode */
  setPreview(preview: boolean): void;

  /** Check equality */
  equals(other: ItemHandle): boolean;
}

// ============================================================================
// Base Item Implementation Helper
// ============================================================================

/**
 * Abstract base class for implementing items.
 * Provides common functionality and state management.
 */
export abstract class BaseItem implements Item {
  readonly id: ItemId;
  readonly type: ItemType;

  protected _tabContent: TabContent;
  protected tabContentListeners: Set<(content: TabContent) => void> = new Set();
  protected dirtyListeners: Set<(isDirty: boolean) => void> = new Set();
  protected changeListeners: Set<(event: ItemChangeEvent) => void> = new Set();
  protected navigateListeners: Set<(event: ItemNavigationEvent) => void> =
    new Set();
  protected _focusHandle: FocusHandle | null = null;
  protected disposed = false;

  constructor(
    id: ItemId,
    type: ItemType,
    initialTabContent: Partial<TabContent> = {}
  ) {
    this.id = id;
    this.type = type;

    this._tabContent = {
      title: "Untitled",
      isDirty: false,
      isPreview: false,
      isPinned: false,
      ...initialTabContent,
    };
  }

  get focusHandle(): FocusHandle {
    if (!this._focusHandle) {
      throw new Error("Focus handle not initialized");
    }
    return this._focusHandle;
  }

  // Tab Content
  getTabContent(): TabContent {
    return this._tabContent;
  }

  onTabContentChange(callback: (content: TabContent) => void): Subscription {
    this.tabContentListeners.add(callback);
    return {
      id: crypto.randomUUID(),
      unsubscribe: () => this.tabContentListeners.delete(callback),
    };
  }

  protected updateTabContent(update: Partial<TabContent>): void {
    const prevDirty = this._tabContent.isDirty;
    this._tabContent = { ...this._tabContent, ...update };

    // Notify tab content listeners
    for (const listener of this.tabContentListeners) {
      listener(this._tabContent);
    }

    // Notify dirty listeners if dirty changed
    if (update.isDirty !== undefined && update.isDirty !== prevDirty) {
      for (const listener of this.dirtyListeners) {
        listener(update.isDirty);
      }
    }
  }

  // Project Path - override in file-based items
  getProjectPath(): ProjectPath | null {
    return null;
  }

  getDeduplicationKey(): string | null {
    return null;
  }

  // Dirty State
  isDirty(): boolean {
    return this._tabContent.isDirty;
  }

  hasConflict(): boolean {
    return false;
  }

  onDirtyChange(callback: (isDirty: boolean) => void): Subscription {
    this.dirtyListeners.add(callback);
    return {
      id: crypto.randomUUID(),
      unsubscribe: () => this.dirtyListeners.delete(callback),
    };
  }

  protected setDirty(isDirty: boolean): void {
    this.updateTabContent({ isDirty });
  }

  // Persistence - override in saveable items
  canSave(): boolean {
    return false;
  }

  async save(_token?: CancellationToken): Promise<Result<void>> {
    return { ok: false, error: new Error("Save not supported") };
  }

  canSaveAs(): boolean {
    return false;
  }

  async saveAs(
    _path: string,
    _token?: CancellationToken
  ): Promise<Result<void>> {
    return { ok: false, error: new Error("Save As not supported") };
  }

  async reload(_token?: CancellationToken): Promise<Result<void>> {
    return { ok: false, error: new Error("Reload not supported") };
  }

  // Splitting - override in splittable items
  canSplit(): boolean {
    return false;
  }

  cloneForSplit(): Item | null {
    return null;
  }

  isSingleton(): boolean {
    return false;
  }

  // Focus
  focus(): void {
    this._focusHandle?.focus();
  }

  hasFocus(): boolean {
    return this._focusHandle?.isFocused() ?? false;
  }

  // Events
  onChange(callback: (event: ItemChangeEvent) => void): Subscription {
    this.changeListeners.add(callback);
    return {
      id: crypto.randomUUID(),
      unsubscribe: () => this.changeListeners.delete(callback),
    };
  }

  onNavigate(callback: (event: ItemNavigationEvent) => void): Subscription {
    this.navigateListeners.add(callback);
    return {
      id: crypto.randomUUID(),
      unsubscribe: () => this.navigateListeners.delete(callback),
    };
  }

  protected emitChange(event: Omit<ItemChangeEvent, "itemId">): void {
    const fullEvent = { ...event, itemId: this.id };
    for (const listener of this.changeListeners) {
      listener(fullEvent);
    }
  }

  protected emitNavigate(event: ItemNavigationEvent): void {
    for (const listener of this.navigateListeners) {
      listener(event);
    }
  }

  // Serialization - override for persistent items
  serialize(): ItemSerializedState | null {
    return null;
  }

  // Abstract method - must implement
  abstract render(): ReactNode;

  // Optional lifecycle hooks - override in subclasses
  onActivate?(): void;
  onDeactivate?(): void;
  onWillClose?(): boolean | Promise<boolean>;
  onClose?(): void;

  // Cleanup
  dispose(): void {
    if (!this.disposed) {
      this.disposed = true;
      this.tabContentListeners.clear();
      this.dirtyListeners.clear();
      this.changeListeners.clear();
      this.navigateListeners.clear();
      this.onClose?.();
    }
  }
}
