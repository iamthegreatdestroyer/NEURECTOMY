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
