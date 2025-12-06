/**
 * @neurectomy/api-client - GraphQL & REST API Client
 *
 * Provides type-safe API clients for communicating with Neurectomy services,
 * including GraphQL queries/mutations and REST endpoints.
 */

// Export GraphQL client
export {
  GraphQLClient,
  createGraphQLClient,
  type GraphQLClientConfig,
} from "./graphql";

// Export Subscription Manager
export {
  SubscriptionManager,
  createSubscriptionManager,
  type SubscriptionManagerConfig,
  type SubscriptionOptions as SubscriptionManagerOptions,
  type SubscriptionFilter,
  type ConnectionStateEvent,
  type SubscriptionConnectionState,
  type ThrottleStrategy,
} from "./graphql";

// Export Schema Governance
export {
  // Schema Registry
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
  // Migration Tools
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
  type MigrationToolError,
  type QueryTransformation,
  type TransformResult,
  // Deprecation Tracking
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
} from "./graphql";

// Export Connection Management
export {
  // Connection State
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
  // Heartbeat
  HeartbeatHandler,
  createHeartbeatHandler,
  createIntegratedKeepalive,
  type HeartbeatConfig,
  type HeartbeatStats,
  type HeartbeatEvent,
  type PingMessage,
  type PongMessage,
  type IntegratedKeepaliveOptions,
} from "./graphql";

// Export REST client
export { RestClient, createRestClient, type RestClientConfig } from "./rest";

// Export hooks for React integration
export {
  useQuery,
  useMutation,
  useSubscription,
  type QueryOptions,
  type MutationOptions,
  type SubscriptionOptions,
} from "./hooks";

// Export utilities
export { handleApiError, retryRequest, type ApiError } from "./utils";

// Export versioning utilities
export {
  buildVersionedUrl,
  getVersionHeaders,
  parseVersion,
  isVersionSupported,
  checkVersionCompatibility,
  getMigrationPath,
  DEFAULT_API_VERSION,
  API_VERSION_HEADER,
  VERSION_MIGRATIONS,
  type ApiVersion,
  type VersionedApiConfig,
  type VersionCompatibility,
  type VersionMigration,
} from "./versioning";

// Re-export types
export type { Agent, Task, Workflow, ExecutionResult } from "./types";
