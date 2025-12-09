/**
 * Provider Interface
 *
 * Defines contracts for AI context providers, completions, and embeddings.
 * Based on Continue's provider patterns adapted for TypeScript.
 *
 * Providers supply context and capabilities to the AI system.
 *
 * @module @neurectomy/interfaces
 * @author @APEX @ARCHITECT
 */

import type {
  EntityId,
  Disposable,
  Subscription,
  CancellationToken,
  Result,
  Range,
  Position,
} from "./types";

// ============================================================================
// Provider Identity
// ============================================================================

/**
 * Context provider ID type
 */
export type ContextProviderId = EntityId<"context">;

/**
 * Built-in context provider identifiers
 */
export const ContextProviderIds = {
  FILE: "context:file" as ContextProviderId,
  CODE: "context:code" as ContextProviderId,
  FOLDER: "context:folder" as ContextProviderId,
  TERMINAL: "context:terminal" as ContextProviderId,
  DIFF: "context:diff" as ContextProviderId,
  PROBLEMS: "context:problems" as ContextProviderId,
  DOCS: "context:docs" as ContextProviderId,
  WEB: "context:web" as ContextProviderId,
  SEARCH: "context:search" as ContextProviderId,
  CODEBASE: "context:codebase" as ContextProviderId,
  OPEN_FILES: "context:open-files" as ContextProviderId,
  GIT: "context:git" as ContextProviderId,
} as const;

// ============================================================================
// Context Items
// ============================================================================

/**
 * A piece of context to include in AI prompts.
 * This is the core data structure for context providers.
 */
export interface ContextItem {
  /** Unique identifier for this context item */
  id: string;

  /** Human-readable name */
  name: string;

  /** Description of what this context contains */
  description: string;

  /** The actual content to include in prompts */
  content: string;

  /** URI for the source of this context */
  uri?: string;

  /** Range within the source (for code snippets) */
  range?: Range;

  /** Language identifier (for syntax highlighting) */
  language?: string;

  /** Whether this is editable context */
  editable?: boolean;

  /** Icon identifier for display */
  icon?: string;

  /** Priority for ordering (higher = more important) */
  priority?: number;

  /** Token count estimate */
  tokenCount?: number;

  /** Metadata for additional information */
  metadata?: Record<string, unknown>;
}

/**
 * Submenu item for hierarchical context selection
 */
export interface ContextSubmenuItem {
  /** Unique identifier */
  id: string;

  /** Display label */
  label: string;

  /** Description */
  description?: string;

  /** Icon identifier */
  icon?: string;

  /** Nested items (if this is a folder-like item) */
  children?: ContextSubmenuItem[];
}

// ============================================================================
// Context Provider Interface
// ============================================================================

/**
 * Provider description for registration and display
 */
export interface ContextProviderDescription {
  /** Unique provider ID */
  id: ContextProviderId;

  /** Human-readable display name */
  displayName: string;

  /** Short description */
  description: string;

  /** Icon identifier */
  icon?: string;

  /** Whether provider supports submenu selection */
  supportsSubmenu?: boolean;

  /** Whether provider requires query input */
  requiresQuery?: boolean;

  /** Trigger characters that activate this provider (e.g., "@file") */
  triggerCharacters?: string[];
}

/**
 * Parameters for fetching context items
 */
export interface ContextFetchParams {
  /** Query string for filtering/searching */
  query?: string;

  /** Selected submenu item path */
  selectedPath?: string[];

  /** Maximum number of items to return */
  limit?: number;

  /** Cancellation token */
  token?: CancellationToken;

  /** Additional provider-specific options */
  options?: Record<string, unknown>;
}

/**
 * Core context provider interface.
 *
 * Based on Continue's BaseContextProvider:
 * ```typescript
 * abstract class BaseContextProvider {
 *   abstract getContextItems(query: string): Promise<ContextItem[]>;
 *   loadSubmenuItems?(args: LoadSubmenuItemsArgs): Promise<ContextSubmenuItem[]>;
 * }
 * ```
 */
export interface ContextProvider extends Disposable {
  /** Provider description */
  readonly description: ContextProviderDescription;

  /**
   * Get context items for inclusion in prompts.
   * This is the primary method for fetching context.
   */
  getContextItems(params: ContextFetchParams): Promise<ContextItem[]>;

  /**
   * Load submenu items for hierarchical selection.
   * Called when user is browsing context options.
   */
  loadSubmenuItems?(
    path: string[],
    token?: CancellationToken
  ): Promise<ContextSubmenuItem[]>;

  /**
   * Validate if this provider is available/configured.
   */
  isAvailable(): boolean | Promise<boolean>;

  /**
   * Called when provider is first activated.
   */
  activate?(): Promise<void>;
}

// ============================================================================
// Completion Types
// ============================================================================

/**
 * A code completion suggestion
 */
export interface CompletionItem {
  /** The text to insert */
  insertText: string;

  /** Display label */
  label: string;

  /** Position where completion starts */
  range: Range;

  /** Completion kind (for icon) */
  kind: CompletionKind;

  /** Documentation */
  documentation?: string;

  /** Detail text (shown after label) */
  detail?: string;

  /** Sort order */
  sortText?: string;

  /** Filter text (for matching) */
  filterText?: string;

  /** Whether this is a snippet */
  isSnippet?: boolean;

  /** Confidence score (0-1) */
  score?: number;

  /** Provider that generated this completion */
  provider?: string;
}

/**
 * Completion kind for categorization
 */
export enum CompletionKind {
  Text = 1,
  Method = 2,
  Function = 3,
  Constructor = 4,
  Field = 5,
  Variable = 6,
  Class = 7,
  Interface = 8,
  Module = 9,
  Property = 10,
  Unit = 11,
  Value = 12,
  Enum = 13,
  Keyword = 14,
  Snippet = 15,
  Color = 16,
  File = 17,
  Reference = 18,
  Folder = 19,
  EnumMember = 20,
  Constant = 21,
  Struct = 22,
  Event = 23,
  Operator = 24,
  TypeParameter = 25,
}

/**
 * Inline completion (ghost text)
 */
export interface InlineCompletion {
  /** The text to insert */
  text: string;

  /** Range to replace (usually cursor position) */
  range: Range;

  /** Completion command to execute after accepting */
  command?: {
    id: string;
    title: string;
    arguments?: unknown[];
  };
}

// ============================================================================
// Completion Provider Interface
// ============================================================================

/**
 * Parameters for fetching completions
 */
export interface CompletionParams {
  /** Document URI */
  uri: string;

  /** Cursor position */
  position: Position;

  /** Prefix text before cursor */
  prefix: string;

  /** Suffix text after cursor */
  suffix: string;

  /** Full document content */
  content: string;

  /** Language identifier */
  language: string;

  /** Trigger character (if triggered by typing) */
  triggerCharacter?: string;

  /** Whether this is manual completion (Ctrl+Space) */
  isManual?: boolean;

  /** Cancellation token */
  token?: CancellationToken;

  /** Additional context from context providers */
  context?: ContextItem[];
}

/**
 * Completion provider configuration
 */
export interface CompletionProviderConfig {
  /** Provider ID */
  id: string;

  /** Model to use */
  model: string;

  /** API endpoint */
  apiEndpoint?: string;

  /** API key (from secure storage) */
  apiKeyId?: string;

  /** Max tokens to generate */
  maxTokens?: number;

  /** Temperature (0-1) */
  temperature?: number;

  /** Stop sequences */
  stopSequences?: string[];

  /** Debounce delay in ms */
  debounceMs?: number;

  /** Cache TTL in ms */
  cacheTtlMs?: number;

  /** Whether to enable streaming */
  streaming?: boolean;
}

/**
 * Completion provider interface.
 *
 * Based on Continue's CompletionProvider pattern with:
 * - LRU cache for deduplication
 * - Debouncing for performance
 * - Streaming support
 */
export interface CompletionProvider extends Disposable {
  /** Provider configuration */
  readonly config: CompletionProviderConfig;

  /**
   * Get code completions at cursor position.
   */
  getCompletions(params: CompletionParams): Promise<CompletionItem[]>;

  /**
   * Get inline completions (ghost text).
   * Returns a generator for streaming support.
   */
  getInlineCompletions(
    params: CompletionParams
  ): AsyncGenerator<InlineCompletion, void, unknown>;

  /**
   * Check if provider is ready.
   */
  isReady(): boolean | Promise<boolean>;

  /**
   * Cancel pending requests.
   */
  cancel(): void;

  /**
   * Clear completion cache.
   */
  clearCache(): void;

  /**
   * Subscribe to completion events.
   */
  onCompletion(callback: (items: CompletionItem[]) => void): Subscription;
}

// ============================================================================
// Embeddings Types
// ============================================================================

/**
 * Document chunk for embedding
 */
export interface DocumentChunk {
  /** Unique chunk ID */
  id: string;

  /** Source document URI */
  uri: string;

  /** Content to embed */
  content: string;

  /** Range in source document */
  range?: Range;

  /** Language identifier */
  language?: string;

  /** Chunk index within document */
  index: number;

  /** Metadata for filtering */
  metadata?: Record<string, unknown>;
}

/**
 * Embedding vector result
 */
export interface EmbeddingResult {
  /** Chunk ID */
  chunkId: string;

  /** Embedding vector */
  vector: number[];

  /** Model used */
  model: string;

  /** Token count */
  tokenCount: number;
}

/**
 * Search result from embedding index
 */
export interface EmbeddingSearchResult {
  /** The matching chunk */
  chunk: DocumentChunk;

  /** Similarity score (0-1) */
  score: number;

  /** Distance metric value */
  distance?: number;
}

// ============================================================================
// Embeddings Provider Interface
// ============================================================================

/**
 * Embeddings provider configuration
 */
export interface EmbeddingsProviderConfig {
  /** Provider ID */
  id: string;

  /** Embedding model */
  model: string;

  /** Vector dimension */
  dimension: number;

  /** API endpoint */
  apiEndpoint?: string;

  /** API key ID */
  apiKeyId?: string;

  /** Batch size for embedding */
  batchSize?: number;

  /** Max tokens per chunk */
  maxChunkTokens?: number;
}

/**
 * Embeddings provider interface.
 *
 * Based on Continue's embeddings patterns for RAG.
 */
export interface EmbeddingsProvider extends Disposable {
  /** Provider configuration */
  readonly config: EmbeddingsProviderConfig;

  /**
   * Generate embeddings for document chunks.
   */
  embed(
    chunks: DocumentChunk[],
    token?: CancellationToken
  ): Promise<EmbeddingResult[]>;

  /**
   * Check if provider is ready.
   */
  isReady(): boolean | Promise<boolean>;
}

// ============================================================================
// Embeddings Index Interface
// ============================================================================

/**
 * Index configuration
 */
export interface EmbeddingsIndexConfig {
  /** Index name */
  name: string;

  /** Vector dimension */
  dimension: number;

  /** Similarity metric */
  metric: "cosine" | "euclidean" | "dot";

  /** Storage path */
  storagePath?: string;
}

/**
 * Embeddings index for semantic search.
 *
 * Based on Continue's LanceDbIndex pattern.
 */
export interface EmbeddingsIndex extends Disposable {
  /** Index configuration */
  readonly config: EmbeddingsIndexConfig;

  /**
   * Add embeddings to the index.
   */
  add(results: EmbeddingResult[]): Promise<void>;

  /**
   * Remove embeddings by chunk IDs.
   */
  remove(chunkIds: string[]): Promise<void>;

  /**
   * Search for similar embeddings.
   */
  search(
    query: number[],
    limit: number,
    filter?: Record<string, unknown>
  ): Promise<EmbeddingSearchResult[]>;

  /**
   * Get index statistics.
   */
  getStats(): Promise<{
    totalChunks: number;
    totalTokens: number;
    lastUpdated: Date | null;
  }>;

  /**
   * Clear all entries.
   */
  clear(): Promise<void>;

  /**
   * Check if index is ready.
   */
  isReady(): boolean | Promise<boolean>;
}

// ============================================================================
// Chat/Conversation Types
// ============================================================================

/**
 * Chat message roles
 */
export type ChatRole = "system" | "user" | "assistant" | "tool";

/**
 * A chat message
 */
export interface ChatMessage {
  /** Unique message ID */
  id: string;

  /** Message role */
  role: ChatRole;

  /** Message content */
  content: string;

  /** Timestamp */
  timestamp: Date;

  /** Context items attached to this message */
  context?: ContextItem[];

  /** Tool calls (for assistant messages) */
  toolCalls?: ToolCall[];

  /** Tool result (for tool messages) */
  toolResult?: unknown;

  /** Whether message is still streaming */
  isStreaming?: boolean;

  /** Token count */
  tokenCount?: number;
}

/**
 * Tool call request
 */
export interface ToolCall {
  /** Tool call ID */
  id: string;

  /** Tool name */
  name: string;

  /** Tool arguments */
  arguments: Record<string, unknown>;
}

/**
 * Tool definition
 */
export interface ToolDefinition {
  /** Tool name */
  name: string;

  /** Description for the model */
  description: string;

  /** JSON schema for parameters */
  parameters: Record<string, unknown>;

  /** Execute the tool */
  execute: (args: Record<string, unknown>) => Promise<unknown>;
}

// ============================================================================
// Chat Provider Interface
// ============================================================================

/**
 * Chat provider configuration
 */
export interface ChatProviderConfig {
  /** Provider ID */
  id: string;

  /** Chat model */
  model: string;

  /** API endpoint */
  apiEndpoint?: string;

  /** API key ID */
  apiKeyId?: string;

  /** System prompt */
  systemPrompt?: string;

  /** Max tokens to generate */
  maxTokens?: number;

  /** Temperature (0-1) */
  temperature?: number;

  /** Available tools */
  tools?: ToolDefinition[];
}

/**
 * Chat provider interface.
 */
export interface ChatProvider extends Disposable {
  /** Provider configuration */
  readonly config: ChatProviderConfig;

  /**
   * Send a message and get a streaming response.
   */
  chat(
    messages: ChatMessage[],
    token?: CancellationToken
  ): AsyncGenerator<string, void, unknown>;

  /**
   * Check if provider is ready.
   */
  isReady(): boolean | Promise<boolean>;

  /**
   * Cancel pending requests.
   */
  cancel(): void;

  /**
   * Subscribe to message events.
   */
  onMessage(callback: (message: ChatMessage) => void): Subscription;
}

// ============================================================================
// Provider Registry
// ============================================================================

/**
 * Central registry for all providers
 */
export interface ProviderRegistry {
  // Context Providers
  registerContextProvider(provider: ContextProvider): Disposable;
  getContextProvider(id: ContextProviderId): ContextProvider | undefined;
  getAllContextProviders(): readonly ContextProvider[];

  // Completion Provider
  setCompletionProvider(provider: CompletionProvider): Disposable;
  getCompletionProvider(): CompletionProvider | undefined;

  // Embeddings Provider
  setEmbeddingsProvider(provider: EmbeddingsProvider): Disposable;
  getEmbeddingsProvider(): EmbeddingsProvider | undefined;

  // Chat Provider
  setChatProvider(provider: ChatProvider): Disposable;
  getChatProvider(): ChatProvider | undefined;

  // Events
  onProvidersChange(callback: () => void): Subscription;
}
