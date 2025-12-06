/**
 * @fileoverview Persisted Queries Module for Neurectomy GraphQL Client
 * @module @neurectomy/api-client/persisted-queries
 *
 * Implements persisted queries for improved performance and security:
 * - Query hashing with SHA-256
 * - Client-side query registration
 * - Automatic Persisted Query (APQ) protocol support
 * - Pre-registered common queries for optimal caching
 *
 * Benefits:
 * - Reduced bandwidth (send hash instead of full query)
 * - Improved security (whitelist allowed queries)
 * - Better caching at CDN/edge level
 * - Faster query execution on server
 */

import { createHash } from "crypto";

// ============================================================================
// Types & Interfaces
// ============================================================================

/**
 * Persisted query metadata
 */
export interface PersistedQuery {
  /** SHA-256 hash of the query */
  hash: string;
  /** Original query string */
  query: string;
  /** Human-readable operation name */
  operationName?: string;
  /** Query category for organization */
  category?: QueryCategory;
  /** Description of what the query does */
  description?: string;
  /** Variables schema for validation */
  variableTypes?: Record<string, string>;
  /** Cache TTL hint in seconds */
  cacheTTL?: number;
  /** Whether query is safe to retry */
  idempotent?: boolean;
}

/**
 * Query categories for organization
 */
export type QueryCategory =
  | "agent"
  | "workflow"
  | "conversation"
  | "knowledge"
  | "system"
  | "user"
  | "analytics"
  | "admin";

/**
 * APQ extension format for GraphQL requests
 */
export interface APQExtension {
  persistedQuery: {
    version: 1;
    sha256Hash: string;
  };
}

/**
 * Configuration for the persisted query store
 */
export interface PersistedQueryStoreConfig {
  /** Enable APQ protocol (send hash first, query on miss) */
  enableAPQ?: boolean;
  /** Pre-register common queries on initialization */
  preregisterCommon?: boolean;
  /** Custom hash function (defaults to SHA-256) */
  hashFunction?: (query: string) => string;
  /** Event handler for store events */
  onEvent?: (event: PersistedQueryEvent) => void;
}

/**
 * Events emitted by the store
 */
export type PersistedQueryEvent =
  | { type: "registered"; query: PersistedQuery }
  | { type: "hit"; hash: string }
  | { type: "miss"; hash: string }
  | { type: "apq_retry"; hash: string };

/**
 * Result of APQ execution
 */
export interface APQResult<T = unknown> {
  /** Query execution result */
  data?: T;
  /** Errors from execution */
  errors?: Array<{ message: string; extensions?: Record<string, unknown> }>;
  /** Whether APQ miss occurred and full query was sent */
  apqMiss?: boolean;
}

// ============================================================================
// Hash Utilities
// ============================================================================

/**
 * Generate SHA-256 hash of a query string
 */
export function hashQuery(query: string): string {
  // Normalize query by removing excess whitespace
  const normalized = normalizeQuery(query);
  return createHash("sha256").update(normalized).digest("hex");
}

/**
 * Normalize query string for consistent hashing
 */
export function normalizeQuery(query: string): string {
  return query
    .replace(/\s+/g, " ") // Collapse whitespace
    .replace(/\s*([{}(),:])\s*/g, "$1") // Remove space around punctuation
    .replace(/\s*#[^\n]*\n/g, "") // Remove comments
    .trim();
}

/**
 * Extract operation name from query string
 */
export function extractOperationName(query: string): string | undefined {
  const match = query.match(/(?:query|mutation|subscription)\s+(\w+)/);
  return match?.[1];
}

// ============================================================================
// Persisted Query Store
// ============================================================================

/**
 * Store for managing persisted queries
 */
export class PersistedQueryStore {
  private queries: Map<string, PersistedQuery> = new Map();
  private reverseIndex: Map<string, string> = new Map(); // operationName -> hash
  private config: Required<PersistedQueryStoreConfig>;
  private stats = {
    hits: 0,
    misses: 0,
    registrations: 0,
  };

  constructor(config: PersistedQueryStoreConfig = {}) {
    this.config = {
      enableAPQ: config.enableAPQ ?? true,
      preregisterCommon: config.preregisterCommon ?? true,
      hashFunction: config.hashFunction ?? hashQuery,
      onEvent: config.onEvent ?? (() => {}),
    };

    if (this.config.preregisterCommon) {
      this.registerCommonQueries();
    }
  }

  /**
   * Register a query in the store
   */
  register(
    query: string,
    metadata?: Partial<Omit<PersistedQuery, "hash" | "query">>
  ): PersistedQuery {
    const hash = this.config.hashFunction(query);
    const operationName =
      metadata?.operationName ?? extractOperationName(query);

    const persistedQuery: PersistedQuery = {
      hash,
      query,
      operationName,
      ...metadata,
    };

    this.queries.set(hash, persistedQuery);
    if (operationName) {
      this.reverseIndex.set(operationName, hash);
    }

    this.stats.registrations++;
    this.config.onEvent({ type: "registered", query: persistedQuery });

    return persistedQuery;
  }

  /**
   * Register multiple queries at once
   */
  registerBatch(
    queries: Array<{
      query: string;
      metadata?: Partial<Omit<PersistedQuery, "hash" | "query">>;
    }>
  ): PersistedQuery[] {
    return queries.map(({ query, metadata }) => this.register(query, metadata));
  }

  /**
   * Get query by hash
   */
  getByHash(hash: string): PersistedQuery | undefined {
    const query = this.queries.get(hash);
    if (query) {
      this.stats.hits++;
      this.config.onEvent({ type: "hit", hash });
    } else {
      this.stats.misses++;
      this.config.onEvent({ type: "miss", hash });
    }
    return query;
  }

  /**
   * Get query by operation name
   */
  getByOperationName(operationName: string): PersistedQuery | undefined {
    const hash = this.reverseIndex.get(operationName);
    return hash ? this.getByHash(hash) : undefined;
  }

  /**
   * Check if query is registered
   */
  has(hashOrOperationName: string): boolean {
    return (
      this.queries.has(hashOrOperationName) ||
      this.reverseIndex.has(hashOrOperationName)
    );
  }

  /**
   * Get APQ extension for a query
   */
  getAPQExtension(query: string): APQExtension {
    const hash = this.config.hashFunction(query);
    return {
      persistedQuery: {
        version: 1,
        sha256Hash: hash,
      },
    };
  }

  /**
   * Get all registered queries
   */
  getAllQueries(): PersistedQuery[] {
    return Array.from(this.queries.values());
  }

  /**
   * Get queries by category
   */
  getByCategory(category: QueryCategory): PersistedQuery[] {
    return this.getAllQueries().filter((q) => q.category === category);
  }

  /**
   * Get store statistics
   */
  getStats(): typeof this.stats & { hitRate: number; totalQueries: number } {
    const total = this.stats.hits + this.stats.misses;
    return {
      ...this.stats,
      hitRate: total > 0 ? this.stats.hits / total : 0,
      totalQueries: this.queries.size,
    };
  }

  /**
   * Clear all registered queries
   */
  clear(): void {
    this.queries.clear();
    this.reverseIndex.clear();
    this.stats = { hits: 0, misses: 0, registrations: 0 };
  }

  /**
   * Export queries for server synchronization
   */
  export(): Array<{ hash: string; query: string; operationName?: string }> {
    return this.getAllQueries().map((q) => ({
      hash: q.hash,
      query: q.query,
      operationName: q.operationName,
    }));
  }

  /**
   * Import queries from server/file
   */
  import(
    queries: Array<{ hash: string; query: string; operationName?: string }>
  ): void {
    for (const { hash, query, operationName } of queries) {
      const persistedQuery: PersistedQuery = { hash, query, operationName };
      this.queries.set(hash, persistedQuery);
      if (operationName) {
        this.reverseIndex.set(operationName, hash);
      }
    }
  }

  /**
   * Register common queries used throughout the application
   */
  private registerCommonQueries(): void {
    // Agent queries
    this.register(COMMON_QUERIES.GET_AGENT, {
      operationName: "GetAgent",
      category: "agent",
      description: "Fetch agent by ID with capabilities",
      cacheTTL: 300,
      idempotent: true,
    });

    this.register(COMMON_QUERIES.LIST_AGENTS, {
      operationName: "ListAgents",
      category: "agent",
      description: "List all agents with filtering",
      cacheTTL: 60,
      idempotent: true,
    });

    this.register(COMMON_QUERIES.GET_AGENT_STATUS, {
      operationName: "GetAgentStatus",
      category: "agent",
      description: "Get real-time agent status",
      cacheTTL: 5,
      idempotent: true,
    });

    // Workflow queries
    this.register(COMMON_QUERIES.GET_WORKFLOW, {
      operationName: "GetWorkflow",
      category: "workflow",
      description: "Fetch workflow definition by ID",
      cacheTTL: 300,
      idempotent: true,
    });

    this.register(COMMON_QUERIES.LIST_WORKFLOW_EXECUTIONS, {
      operationName: "ListWorkflowExecutions",
      category: "workflow",
      description: "List workflow executions with filtering",
      cacheTTL: 30,
      idempotent: true,
    });

    // Conversation queries
    this.register(COMMON_QUERIES.GET_CONVERSATION, {
      operationName: "GetConversation",
      category: "conversation",
      description: "Fetch conversation with messages",
      cacheTTL: 60,
      idempotent: true,
    });

    this.register(COMMON_QUERIES.LIST_CONVERSATIONS, {
      operationName: "ListConversations",
      category: "conversation",
      description: "List user conversations",
      cacheTTL: 30,
      idempotent: true,
    });

    // Knowledge base queries
    this.register(COMMON_QUERIES.SEARCH_KNOWLEDGE, {
      operationName: "SearchKnowledge",
      category: "knowledge",
      description: "Semantic search in knowledge base",
      cacheTTL: 60,
      idempotent: true,
    });

    this.register(COMMON_QUERIES.GET_DOCUMENT, {
      operationName: "GetDocument",
      category: "knowledge",
      description: "Fetch document by ID",
      cacheTTL: 300,
      idempotent: true,
    });

    // System queries
    this.register(COMMON_QUERIES.GET_SYSTEM_HEALTH, {
      operationName: "GetSystemHealth",
      category: "system",
      description: "Get system health status",
      cacheTTL: 10,
      idempotent: true,
    });

    this.register(COMMON_QUERIES.GET_METRICS, {
      operationName: "GetMetrics",
      category: "analytics",
      description: "Fetch system metrics",
      cacheTTL: 30,
      idempotent: true,
    });

    // User queries
    this.register(COMMON_QUERIES.GET_CURRENT_USER, {
      operationName: "GetCurrentUser",
      category: "user",
      description: "Get authenticated user profile",
      cacheTTL: 300,
      idempotent: true,
    });
  }
}

// ============================================================================
// Common Query Definitions
// ============================================================================

/**
 * Pre-defined common queries for registration
 */
export const COMMON_QUERIES = {
  // Agent queries
  GET_AGENT: `
    query GetAgent($id: ID!) {
      agent(id: $id) {
        id
        name
        description
        status
        capabilities {
          id
          name
          description
          parameters
        }
        configuration
        createdAt
        updatedAt
      }
    }
  `,

  LIST_AGENTS: `
    query ListAgents($filter: AgentFilterInput, $pagination: PaginationInput) {
      agents(filter: $filter, pagination: $pagination) {
        edges {
          node {
            id
            name
            status
            capabilities {
              id
              name
            }
          }
          cursor
        }
        pageInfo {
          hasNextPage
          hasPreviousPage
          startCursor
          endCursor
          totalCount
        }
      }
    }
  `,

  GET_AGENT_STATUS: `
    query GetAgentStatus($id: ID!) {
      agent(id: $id) {
        id
        status
        metrics {
          executionCount
          successRate
          averageLatency
        }
        lastActivity
      }
    }
  `,

  // Workflow queries
  GET_WORKFLOW: `
    query GetWorkflow($id: ID!) {
      workflow(id: $id) {
        id
        name
        description
        version
        status
        nodes {
          id
          type
          configuration
        }
        edges {
          source
          target
          condition
        }
        createdAt
        updatedAt
      }
    }
  `,

  LIST_WORKFLOW_EXECUTIONS: `
    query ListWorkflowExecutions($workflowId: ID!, $filter: ExecutionFilterInput, $pagination: PaginationInput) {
      workflowExecutions(workflowId: $workflowId, filter: $filter, pagination: $pagination) {
        edges {
          node {
            id
            status
            startedAt
            completedAt
            error
          }
          cursor
        }
        pageInfo {
          hasNextPage
          totalCount
        }
      }
    }
  `,

  // Conversation queries
  GET_CONVERSATION: `
    query GetConversation($id: ID!, $messageLimit: Int = 50) {
      conversation(id: $id) {
        id
        title
        participants {
          id
          type
          name
        }
        messages(last: $messageLimit) {
          edges {
            node {
              id
              role
              content
              timestamp
              metadata
            }
          }
        }
        createdAt
        updatedAt
      }
    }
  `,

  LIST_CONVERSATIONS: `
    query ListConversations($filter: ConversationFilterInput, $pagination: PaginationInput) {
      conversations(filter: $filter, pagination: $pagination) {
        edges {
          node {
            id
            title
            lastMessage {
              content
              timestamp
            }
            unreadCount
          }
          cursor
        }
        pageInfo {
          hasNextPage
          totalCount
        }
      }
    }
  `,

  // Knowledge queries
  SEARCH_KNOWLEDGE: `
    query SearchKnowledge($query: String!, $options: SearchOptionsInput) {
      searchKnowledge(query: $query, options: $options) {
        results {
          document {
            id
            title
            type
          }
          chunk {
            content
            metadata
          }
          score
          highlights
        }
        totalCount
        searchMetadata {
          queryTime
          embeddingModel
        }
      }
    }
  `,

  GET_DOCUMENT: `
    query GetDocument($id: ID!) {
      document(id: $id) {
        id
        title
        type
        content
        metadata
        chunks {
          id
          content
          embedding
        }
        createdAt
        updatedAt
      }
    }
  `,

  // System queries
  GET_SYSTEM_HEALTH: `
    query GetSystemHealth {
      systemHealth {
        status
        services {
          name
          status
          latency
          lastCheck
        }
        database {
          connected
          latency
          poolSize
        }
        cache {
          connected
          hitRate
          memoryUsage
        }
      }
    }
  `,

  GET_METRICS: `
    query GetMetrics($timeRange: TimeRangeInput!, $metrics: [MetricType!]) {
      metrics(timeRange: $timeRange, metrics: $metrics) {
        name
        dataPoints {
          timestamp
          value
        }
        aggregations {
          min
          max
          avg
          sum
        }
      }
    }
  `,

  // User queries
  GET_CURRENT_USER: `
    query GetCurrentUser {
      me {
        id
        email
        name
        role
        preferences
        permissions
        createdAt
      }
    }
  `,
} as const;

// ============================================================================
// APQ Client Integration
// ============================================================================

/**
 * APQ-enabled request builder
 */
export class APQClient {
  private store: PersistedQueryStore;
  private endpoint: string;
  private headers: Record<string, string>;

  constructor(
    endpoint: string,
    store?: PersistedQueryStore,
    headers: Record<string, string> = {}
  ) {
    this.endpoint = endpoint;
    this.store = store ?? new PersistedQueryStore();
    this.headers = headers;
  }

  /**
   * Execute query with APQ protocol
   * First sends hash only, falls back to full query on miss
   */
  async execute<T = unknown, V = Record<string, unknown>>(
    query: string,
    variables?: V,
    operationName?: string
  ): Promise<APQResult<T>> {
    const extension = this.store.getAPQExtension(query);

    // First request: send hash only
    const hashOnlyResult = await this.sendRequest<T>({
      variables,
      operationName,
      extensions: extension,
    });

    // Check for APQ miss error
    if (this.isAPQMiss(hashOnlyResult)) {
      this.store.config.onEvent({
        type: "apq_retry",
        hash: extension.persistedQuery.sha256Hash,
      });

      // Second request: send full query with hash
      const fullResult = await this.sendRequest<T>({
        query,
        variables,
        operationName,
        extensions: extension,
      });

      return {
        ...fullResult,
        apqMiss: true,
      };
    }

    return hashOnlyResult;
  }

  /**
   * Register a query and get its hash
   */
  register(
    query: string,
    metadata?: Partial<Omit<PersistedQuery, "hash" | "query">>
  ): string {
    return this.store.register(query, metadata).hash;
  }

  /**
   * Get the underlying store
   */
  getStore(): PersistedQueryStore {
    return this.store;
  }

  private async sendRequest<T>(body: {
    query?: string;
    variables?: unknown;
    operationName?: string;
    extensions?: APQExtension;
  }): Promise<APQResult<T>> {
    const response = await fetch(this.endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...this.headers,
      },
      body: JSON.stringify(body),
    });

    return response.json();
  }

  private isAPQMiss(result: APQResult): boolean {
    return (
      result.errors?.some(
        (e) =>
          e.message === "PersistedQueryNotFound" ||
          e.extensions?.code === "PERSISTED_QUERY_NOT_FOUND"
      ) ?? false
    );
  }
}

// ============================================================================
// Factory Functions
// ============================================================================

/**
 * Create a new persisted query store
 */
export function createPersistedQueryStore(
  config?: PersistedQueryStoreConfig
): PersistedQueryStore {
  return new PersistedQueryStore(config);
}

/**
 * Create an APQ-enabled client
 */
export function createAPQClient(
  endpoint: string,
  store?: PersistedQueryStore,
  headers?: Record<string, string>
): APQClient {
  return new APQClient(endpoint, store, headers);
}

/**
 * Generate persisted query manifest for build tooling
 */
export function generateQueryManifest(store: PersistedQueryStore): string {
  const queries = store.export();
  return JSON.stringify(
    {
      version: 1,
      generated: new Date().toISOString(),
      queries: queries.map((q) => ({
        hash: q.hash,
        operationName: q.operationName,
        // Don't include query body in manifest for security
      })),
    },
    null,
    2
  );
}
