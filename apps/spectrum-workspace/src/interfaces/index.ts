/**
 * Interfaces Module
 *
 * Central export point for all interface definitions.
 * These interfaces define the contracts for the IDE architecture
 * following patterns from Zed, VS Code, and Continue.
 *
 * @module @neurectomy/interfaces
 * @author @APEX @ARCHITECT
 */

// ============================================================================
// Core Types
// ============================================================================

export {
  // Entity System
  type EntityId,
  createEntityId,
  EntityIds,

  // Focus Management
  type FocusHandle,
  type FocusContext,

  // Layout Types
  type DockPosition,
  type SplitDirection,
  type Axis,
  type SizeConstraints,
  type Bounds,

  // Subscription System
  type Subscription,
  type Observable,
  type MutableObservable,

  // Action System
  type KeyBinding,
  type Action,
  type ActionContext,
  type WorkspaceContext,

  // File System Types
  type FileEntry,
  type DirectoryEntry,
  type FileChangeType,
  type FileChangeEvent,

  // Diagnostic Types
  DiagnosticSeverity,
  type Position,
  type Range,
  type Diagnostic,
  type DiagnosticRelatedInformation,
  DiagnosticTag,

  // Theme Types
  type ThemeMode,
  type IconTheme,

  // Utility Types
  type DeepPartial,
  type RequireFields,
  type AsyncResult,
  type Result,
  ok,
  err,
  type Disposable,
  DisposableStore,

  // Event Types
  type EventEmitter,
  type CancellationToken,
  type CancellationTokenSource,
} from "./types";

// ============================================================================
// Panel Interface
// ============================================================================

export {
  // Identity
  type PanelId,
  PanelIds,

  // Configuration
  type PanelConfig,

  // State
  type PanelState,
  type PanelStateUpdate,

  // Core Interface
  type Panel,

  // Factory & Registry
  type PanelFactory,
  type PanelRegistration,
  type PanelRegistry,
  type PanelManager,

  // Base Implementation
  BasePanel,
} from "./Panel";

// ============================================================================
// Item Interface
// ============================================================================

export {
  // Identity
  type ItemId,
  type ItemType,

  // Tab Presentation
  type TabContent,
  type TabState,

  // Project Path
  type ProjectPath,

  // Events
  type ItemChangeEvent,
  type ItemNavigationEvent,

  // Core Interface
  type Item,

  // Serialization
  type ItemSerializedState,
  type ItemDeserializer,

  // Factory & Registry
  type ItemCreateParams,
  type ItemFactory,
  type ItemRegistration,
  type ItemRegistry,

  // Handle
  type ItemHandle,

  // Base Implementation
  BaseItem,
} from "./Item";

// ============================================================================
// Provider Interface
// ============================================================================

export {
  // Context Provider Identity
  type ContextProviderId,
  ContextProviderIds,

  // Context Items
  type ContextItem,
  type ContextSubmenuItem,

  // Context Provider
  type ContextProviderDescription,
  type ContextFetchParams,
  type ContextProvider,

  // Completion Types
  type CompletionItem,
  CompletionKind,
  type InlineCompletion,

  // Completion Provider
  type CompletionParams,
  type CompletionProviderConfig,
  type CompletionProvider,

  // Embeddings Types
  type DocumentChunk,
  type EmbeddingResult,
  type EmbeddingSearchResult,

  // Embeddings Provider
  type EmbeddingsProviderConfig,
  type EmbeddingsProvider,

  // Embeddings Index
  type EmbeddingsIndexConfig,
  type EmbeddingsIndex,

  // Chat Types
  type ChatRole,
  type ChatMessage,
  type ToolCall,
  type ToolDefinition,

  // Chat Provider
  type ChatProviderConfig,
  type ChatProvider,

  // Provider Registry
  type ProviderRegistry,
} from "./Provider";

// ============================================================================
// Workspace Interface
// ============================================================================

export {
  // Identity
  type WorkspaceId,
  type PaneId,
  type DockId,

  // Project
  type ProjectConfig,
  type Project,

  // Pane
  type PaneState,
  type Pane,
  type PaneItemChangeEvent,
  type PaneSerializedState,

  // Pane Group
  type PaneGroupNode,
  type PaneGroupLeaf,
  type PaneGroupSplit,
  type PaneGroup,
  type PaneGroupChangeEvent,
  type PaneGroupSerializedState,

  // Dock
  type DockState,
  type Dock,
  type DockSerializedState,

  // Workspace
  type WorkspaceState,
  type Workspace,
  type WorkspaceSerializedState,

  // Factory
  type WorkspaceOptions,
  type WorkspaceFactory,
} from "./Workspace";
