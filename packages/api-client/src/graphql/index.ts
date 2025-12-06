/**
 * =============================================================================
 * GraphQL Client Module
 * =============================================================================
 * Unified exports for all GraphQL client functionality including:
 * - Core GraphQL client with query/mutation/subscription support
 * - Production-resilient subscription manager
 * - Persisted queries with APQ protocol
 * - Type-safe mutation executor
 * =============================================================================
 */

import { GraphQLClient as GQLClient } from "graphql-request";
import type { Variables } from "graphql-request";
import { createClient as createWSClient } from "graphql-ws";
import type { Client as WSClient } from "graphql-ws";

// =============================================================================
// Re-exports from submodules
// =============================================================================

// Subscription Manager
export {
  SubscriptionManager,
  createSubscriptionManager,
  type SubscriptionManagerConfig,
  type SubscriptionOptions,
  type SubscriptionFilter,
  type ConnectionState as SubscriptionConnectionState,
  type ConnectionStateEvent,
  type ThrottleStrategy,
} from "./subscription-manager";

// Persisted Queries
export {
  PersistedQueryStore,
  APQClient,
  createPersistedQueryStore,
  createAPQClient,
  hashQuery,
  normalizeQuery,
  extractOperationName,
  generateQueryManifest,
  COMMON_QUERIES,
  type PersistedQuery,
  type PersistedQueryStoreConfig,
  type QueryCategory,
  type APQExtension,
  type PersistedQueryEvent,
  type APQResult,
} from "./persisted-queries";

// Mutations
export {
  MutationExecutor,
  MutationError,
  createMutationExecutor,
  defineMutation,
  withValidation,
  validateInput,
  executeMutation,
  setGlobalMutationExecutor,
  MUTATIONS,
  type MutationExecutorConfig,
  type MutationOptions,
  type MutationResult,
  type UserError,
  type MutationErrorType,
} from "./mutations";

// Schema Registry & Versioning
export {
  SchemaRegistry,
  SchemaIncompatibilityError,
  createSchemaRegistry,
  createStrictSchemaRegistry,
  formatCompatibilityReport,
  type SchemaRegistryConfig,
  type SchemaRegistryEvents,
  type SchemaVersionInfo,
  type SemanticVersion,
  type ChangelogEntry,
  type ChangeType,
  type BreakingChange,
  type BreakingChangeType,
  type CodeExample,
  type CompatibilityLevel,
  type CompatibilityReport,
  type MigrationEstimate,
  type SchemaElementType,
  type DeprecationInfo,
  type SchemaHealth,
  type ComplexTypeInfo,
} from "./schema-registry";

// Migration Tools
export {
  MigrationTools,
  createMigrationTools,
  formatMigrationGuide,
  formatMigrationResult,
  type MigrationToolsConfig,
  type MigrationToolsEvents,
  type MigrationStep,
  type MigrationStepStatus,
  type Codemod,
  type CodemodLanguage,
  type CodemodTestCase,
  type MigrationGuide,
  type MigrationEffort,
  type PreflightCheck,
  type PreflightStatus,
  type RollbackPlan,
  type RollbackStep,
  type MigrationContext,
  type MigrationResult,
  type MigrationError as MigrationToolError,
  type QueryTransformation,
  type TransformResult,
} from "./migration-tools";

// Deprecation Tracking
export {
  DeprecationTracker,
  createDeprecationTracker,
  formatDeprecationReport,
  formatImpactAnalysis,
  type DeprecationTrackerConfig,
  type DeprecationTrackerEvents,
  type DeprecationEntry,
  type DeprecationState,
  type DeprecationReport,
  type UsageStats,
  type TimeSeriesData,
  type ClientInfo,
  type MigrationStatus,
  type SunsetSchedule,
  type SunsetPhase,
  type NotificationSchedule,
  type NotificationType,
  type AutomaticAction,
  type AutomaticActionType,
  type UsageTrackingEntry,
  type ImpactAnalysis,
} from "./deprecation-tracker";

// Connection State Management
export {
  ConnectionStateManager,
  ConnectionPool,
  createConnectionStateManager,
  createConnectionPool,
  type ConnectionState,
  type ConnectionQuality,
  type CircuitBreakerState,
  type ConnectionStateConfig,
  type ConnectionMetrics,
  type ConnectionDiagnostics,
  type StateTransition,
  type CircuitBreakerConfig,
  type ConnectionEvent,
} from "./connection-state";

// Heartbeat Handler
export {
  HeartbeatHandler,
  createHeartbeatHandler,
  createIntegratedKeepalive,
  type HeartbeatConfig,
  type HeartbeatStats,
  type HeartbeatEvent,
  type PingMessage,
  type PongMessage,
  type IntegratedKeepaliveOptions,
} from "./heartbeat-handler";

// =============================================================================
// Core GraphQL Client
// =============================================================================

export interface GraphQLClientConfig {
  endpoint: string;
  wsEndpoint?: string;
  headers?: Record<string, string>;
  timeout?: number;
}

/**
 * Type-safe GraphQL client with subscription support.
 *
 * @description
 * Provides a unified interface for GraphQL operations:
 * - Queries: Fetch data from the server
 * - Mutations: Modify data on the server
 * - Subscriptions: Real-time data updates via WebSocket
 *
 * @example
 * ```typescript
 * const client = createGraphQLClient({
 *   endpoint: 'http://localhost:4000/graphql',
 *   wsEndpoint: 'ws://localhost:4000/graphql',
 * });
 *
 * // Query
 * const data = await client.query<{ user: User }>(`
 *   query GetUser($id: ID!) { user(id: $id) { id name } }
 * `, { id: '123' });
 *
 * // Mutation
 * const result = await client.mutate<{ createUser: User }>(`
 *   mutation CreateUser($input: CreateUserInput!) {
 *     createUser(input: $input) { id name }
 *   }
 * `, { input: { name: 'John' } });
 *
 * // Subscription
 * const unsubscribe = client.subscribe<{ userUpdated: User }>(`
 *   subscription OnUserUpdated($id: ID!) {
 *     userUpdated(id: $id) { id name }
 *   }
 * `, (data) => console.log(data), { id: '123' });
 * ```
 */
export class GraphQLClient {
  private client: GQLClient;
  private wsClient: WSClient | null = null;
  private config: GraphQLClientConfig;

  constructor(config: GraphQLClientConfig) {
    this.config = config;
    this.client = new GQLClient(config.endpoint, {
      headers: config.headers,
      // Note: timeout is handled via fetch options or AbortController if needed
    });

    if (config.wsEndpoint) {
      this.wsClient = createWSClient({
        url: config.wsEndpoint,
        connectionParams: {
          headers: config.headers,
        },
      });
    }
  }

  /**
   * Execute a GraphQL query.
   *
   * @param document - The GraphQL query document
   * @param variables - Variables for the query
   * @returns The query result data
   */
  async query<TData, TVariables extends Variables = Variables>(
    document: string,
    variables?: TVariables
  ): Promise<TData> {
    return this.client.request<TData>(document, variables);
  }

  /**
   * Execute a GraphQL mutation.
   *
   * @param document - The GraphQL mutation document
   * @param variables - Variables for the mutation
   * @returns The mutation result data
   */
  async mutate<TData, TVariables extends Variables = Variables>(
    document: string,
    variables?: TVariables
  ): Promise<TData> {
    return this.client.request<TData>(document, variables);
  }

  /**
   * Subscribe to a GraphQL subscription.
   *
   * @param document - The GraphQL subscription document
   * @param onData - Callback for received data
   * @param variables - Variables for the subscription
   * @param onError - Optional error callback
   * @returns Unsubscribe function
   *
   * @throws Error if WebSocket client is not configured
   */
  subscribe<TData, TVariables extends Variables = Variables>(
    document: string,
    onData: (data: TData) => void,
    variables?: TVariables,
    onError?: (error: Error) => void
  ): () => void {
    if (!this.wsClient) {
      throw new Error("WebSocket client not configured");
    }

    const unsubscribe = this.wsClient.subscribe<TData>(
      {
        query: document,
        variables: variables as Record<string, unknown>,
      },
      {
        next: (result) => {
          if (result.data) {
            onData(result.data);
          }
        },
        error: (err) => {
          if (onError) {
            onError(err instanceof Error ? err : new Error(String(err)));
          }
        },
        complete: () => {},
      }
    );

    return unsubscribe;
  }

  /**
   * Set authorization header.
   *
   * @param token - The bearer token
   */
  setAuthToken(token: string): void {
    this.client.setHeader("Authorization", `Bearer ${token}`);
  }

  /**
   * Clear authorization header.
   */
  clearAuthToken(): void {
    this.client.setHeader("Authorization", "");
  }

  /**
   * Get the underlying graphql-request client.
   * Useful for advanced use cases.
   */
  getClient(): GQLClient {
    return this.client;
  }

  /**
   * Get the WebSocket client if configured.
   */
  getWSClient(): WSClient | null {
    return this.wsClient;
  }

  /**
   * Dispose of WebSocket connection.
   */
  dispose(): void {
    if (this.wsClient) {
      this.wsClient.dispose();
    }
  }
}

/**
 * Factory function to create a GraphQL client.
 *
 * @param config - Client configuration
 * @returns Configured GraphQL client instance
 *
 * @example
 * ```typescript
 * const client = createGraphQLClient({
 *   endpoint: process.env.GRAPHQL_ENDPOINT!,
 *   wsEndpoint: process.env.GRAPHQL_WS_ENDPOINT,
 *   headers: {
 *     'X-Api-Key': process.env.API_KEY!,
 *   },
 * });
 * ```
 */
export function createGraphQLClient(
  config: GraphQLClientConfig
): GraphQLClient {
  return new GraphQLClient(config);
}

// =============================================================================
// Utility Types
// =============================================================================

/**
 * Extract the data type from a GraphQL operation result.
 */
export type ExtractData<T> = T extends { data: infer D } ? D : never;

/**
 * Extract variables type from a GraphQL operation.
 */
export type ExtractVariables<T> = T extends { variables: infer V } ? V : never;

/**
 * GraphQL operation result with optional errors.
 */
export interface GraphQLResult<TData> {
  data?: TData;
  errors?: GraphQLError[];
}

/**
 * GraphQL error structure.
 */
export interface GraphQLError {
  message: string;
  locations?: Array<{ line: number; column: number }>;
  path?: Array<string | number>;
  extensions?: Record<string, unknown>;
}

// =============================================================================
// Default Export
// =============================================================================

export default GraphQLClient;
