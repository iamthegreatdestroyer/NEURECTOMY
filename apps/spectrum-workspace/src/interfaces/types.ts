/**
 * Core Types & Shared Definitions
 *
 * Foundational types used throughout the IDE architecture,
 * following patterns from Zed (DockPosition, FocusHandle) and VS Code.
 *
 * @module @neurectomy/interfaces
 * @author @APEX @ARCHITECT
 */

// ============================================================================
// Entity System Types (Zed-inspired)
// ============================================================================

/**
 * Unique identifier for entities in the workspace.
 * Uses branded types for type safety.
 */
export type EntityId<T extends string = string> = string & {
  readonly __entity: T;
};

/**
 * Generate a new unique entity ID
 */
export function createEntityId<T extends string>(prefix: T): EntityId<T> {
  return `${prefix}_${crypto.randomUUID()}` as EntityId<T>;
}

/**
 * Type-safe entity ID generators
 */
export const EntityIds = {
  pane: () => createEntityId("pane"),
  panel: () => createEntityId("panel"),
  item: () => createEntityId("item"),
  tab: () => createEntityId("tab"),
  workspace: () => createEntityId("workspace"),
  dock: () => createEntityId("dock"),
  view: () => createEntityId("view"),
  context: () => createEntityId("context"),
} as const;

// ============================================================================
// Focus Management (Zed-inspired)
// ============================================================================

/**
 * Focus handle for keyboard navigation and focus management.
 * Similar to Zed's FocusHandle but adapted for React.
 */
export interface FocusHandle {
  /** Unique identifier for the focus target */
  readonly id: string;

  /** Focus this element */
  focus(): void;

  /** Check if this element is currently focused */
  isFocused(): boolean;

  /** Check if focus is within this element or its descendants */
  containsFocus(): boolean;

  /** Blur this element */
  blur(): void;
}

/**
 * Context for managing focus within a component tree
 */
export interface FocusContext {
  /** Current focus stack - most recent first */
  readonly stack: readonly FocusHandle[];

  /** Request focus for a handle */
  requestFocus(handle: FocusHandle): void;

  /** Release focus from a handle */
  releaseFocus(handle: FocusHandle): void;

  /** Get the currently focused handle */
  current(): FocusHandle | null;
}

// ============================================================================
// Position & Layout Types
// ============================================================================

/**
 * Position for dockable panels (Zed pattern)
 */
export type DockPosition = "left" | "right" | "bottom";

/**
 * Direction for pane splits
 */
export type SplitDirection = "horizontal" | "vertical";

/**
 * Axis type for layout calculations
 */
export type Axis = "x" | "y";

/**
 * Size constraints for resizable elements
 */
export interface SizeConstraints {
  minWidth?: number;
  maxWidth?: number;
  minHeight?: number;
  maxHeight?: number;
}

/**
 * Bounds for positioned elements
 */
export interface Bounds {
  x: number;
  y: number;
  width: number;
  height: number;
}

// ============================================================================
// Subscription System Types (Zed-inspired)
// ============================================================================

/**
 * Subscription handle for reactive updates
 */
export interface Subscription {
  /** Unique subscription ID */
  readonly id: string;

  /** Unsubscribe from updates */
  unsubscribe(): void;
}

/**
 * Observable value that notifies subscribers of changes
 */
export interface Observable<T> {
  /** Get current value */
  get(): T;

  /** Subscribe to changes */
  subscribe(callback: (value: T, previous: T) => void): Subscription;

  /** Map to a derived observable */
  map<U>(mapper: (value: T) => U): Observable<U>;
}

/**
 * Mutable observable that can be updated
 */
export interface MutableObservable<T> extends Observable<T> {
  /** Set new value */
  set(value: T): void;

  /** Update value using a function */
  update(updater: (current: T) => T): void;
}

// ============================================================================
// Action System Types (VS Code Command Pattern)
// ============================================================================

/**
 * Keyboard shortcut binding
 */
export interface KeyBinding {
  /** Primary key combination (e.g., "ctrl+s", "cmd+shift+p") */
  key: string;

  /** When clause for conditional activation */
  when?: string;

  /** Secondary key combinations */
  secondaryKeys?: string[];

  /** macOS-specific override */
  mac?: string;

  /** Windows-specific override */
  win?: string;

  /** Linux-specific override */
  linux?: string;
}

/**
 * Action definition for commands
 */
export interface Action<T = void> {
  /** Unique action identifier */
  readonly id: string;

  /** Human-readable label */
  readonly label: string;

  /** Icon identifier */
  readonly icon?: string;

  /** Keyboard binding */
  readonly keybinding?: KeyBinding;

  /** Execute the action */
  execute(context: ActionContext): Promise<T> | T;

  /** Check if action is enabled in current context */
  isEnabled?(context: ActionContext): boolean;

  /** Check if action is visible in current context */
  isVisible?(context: ActionContext): boolean;
}

/**
 * Context passed to action execution
 */
export interface ActionContext {
  /** Current workspace state */
  readonly workspace: WorkspaceContext;

  /** Current focus context */
  readonly focus: FocusContext;

  /** Selected items/elements */
  readonly selection: readonly unknown[];

  /** Custom context data */
  readonly data: Record<string, unknown>;
}

/**
 * Minimal workspace context for actions
 */
export interface WorkspaceContext {
  /** Current project path */
  readonly projectPath: string | null;

  /** Active pane ID */
  readonly activePaneId: string | null;

  /** Active item ID */
  readonly activeItemId: string | null;

  /** Whether a file is open */
  readonly hasOpenFile: boolean;
}

// ============================================================================
// File System Types
// ============================================================================

/**
 * File entry in the workspace
 */
export interface FileEntry {
  /** Full path to the file */
  path: string;

  /** File name (last segment of path) */
  name: string;

  /** Whether this is a directory */
  isDirectory: boolean;

  /** Whether this is a symbolic link */
  isSymlink: boolean;

  /** File size in bytes (null for directories) */
  size: number | null;

  /** Last modified timestamp */
  modifiedAt: Date;

  /** Created timestamp */
  createdAt: Date;

  /** Whether the file is hidden */
  isHidden: boolean;

  /** File extension (without dot) */
  extension: string | null;
}

/**
 * Directory entry with children
 */
export interface DirectoryEntry extends FileEntry {
  isDirectory: true;

  /** Child entries (lazy loaded) */
  children: FileEntry[] | null;

  /** Whether children have been loaded */
  childrenLoaded: boolean;

  /** Whether this directory is expanded in the UI */
  isExpanded: boolean;
}

/**
 * File change event types
 */
export type FileChangeType = "created" | "modified" | "deleted" | "renamed";

/**
 * File change event
 */
export interface FileChangeEvent {
  type: FileChangeType;
  path: string;
  oldPath?: string; // For renames
}

// ============================================================================
// Diagnostic Types (LSP-aligned)
// ============================================================================

/**
 * Diagnostic severity levels
 */
export enum DiagnosticSeverity {
  Error = 1,
  Warning = 2,
  Information = 3,
  Hint = 4,
}

/**
 * Position in a text document (0-indexed)
 */
export interface Position {
  line: number;
  character: number;
}

/**
 * Range in a text document
 */
export interface Range {
  start: Position;
  end: Position;
}

/**
 * A diagnostic represents a problem in code
 */
export interface Diagnostic {
  /** Range in the document */
  range: Range;

  /** Severity level */
  severity: DiagnosticSeverity;

  /** Short message */
  message: string;

  /** Source of the diagnostic (e.g., "typescript", "eslint") */
  source?: string;

  /** Diagnostic code */
  code?: string | number;

  /** Related information */
  relatedInformation?: DiagnosticRelatedInformation[];

  /** Tags (deprecated, unnecessary) */
  tags?: DiagnosticTag[];
}

/**
 * Related diagnostic information
 */
export interface DiagnosticRelatedInformation {
  location: {
    uri: string;
    range: Range;
  };
  message: string;
}

/**
 * Diagnostic tags
 */
export enum DiagnosticTag {
  Unnecessary = 1,
  Deprecated = 2,
}

// ============================================================================
// Theme Types
// ============================================================================

/**
 * Theme mode
 */
export type ThemeMode = "light" | "dark" | "system";

/**
 * Icon theme definition
 */
export interface IconTheme {
  id: string;
  label: string;
  path: string;
}

// ============================================================================
// Utility Types
// ============================================================================

/**
 * Deep partial type for nested optional properties
 */
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

/**
 * Make specific properties required
 */
export type RequireFields<T, K extends keyof T> = T & Required<Pick<T, K>>;

/**
 * Async result type for operations that can fail
 */
export type AsyncResult<T, E = Error> = Promise<Result<T, E>>;

/**
 * Result type for fallible operations
 */
export type Result<T, E = Error> =
  | { ok: true; value: T }
  | { ok: false; error: E };

/**
 * Create a success result
 */
export function ok<T>(value: T): Result<T, never> {
  return { ok: true, value };
}

/**
 * Create an error result
 */
export function err<E>(error: E): Result<never, E> {
  return { ok: false, error };
}

/**
 * Disposable resource pattern
 */
export interface Disposable {
  dispose(): void;
}

/**
 * Combined disposable for managing multiple resources
 */
export class DisposableStore implements Disposable {
  private readonly disposables: Disposable[] = [];
  private isDisposed = false;

  add<T extends Disposable>(disposable: T): T {
    if (this.isDisposed) {
      disposable.dispose();
    } else {
      this.disposables.push(disposable);
    }
    return disposable;
  }

  dispose(): void {
    if (!this.isDisposed) {
      this.isDisposed = true;
      for (const d of this.disposables) {
        d.dispose();
      }
      this.disposables.length = 0;
    }
  }
}

// ============================================================================
// Event Types
// ============================================================================

/**
 * Event emitter interface
 */
export interface EventEmitter<T> {
  /** Fire the event with data */
  fire(data: T): void;

  /** Subscribe to the event */
  on(callback: (data: T) => void): Subscription;

  /** Subscribe to the event once */
  once(callback: (data: T) => void): Subscription;
}

/**
 * Cancellation token for async operations
 */
export interface CancellationToken {
  readonly isCancellationRequested: boolean;
  onCancellationRequested: (callback: () => void) => Disposable;
}

/**
 * Source for cancellation tokens
 */
export interface CancellationTokenSource extends Disposable {
  readonly token: CancellationToken;
  cancel(): void;
}
